import {useEffect,useState} from 'react'
import { Outlet, useLocation,Navigate} from 'react-router-dom'
import useUserStore from './store/useUserStore'
import { checkUserAuth } from './services/user.service';
import Loader from './utils/Loader';


export const ProtectedRoute = () => {
    const location = useLocation();
    const [isChecking, setIsChecking] = useState(true);

    const {isAuthenticated, setUser,clearUser} = useUserStore();

    useEffect(()=> {
        const varifyAuth = async () => {
            try {
                const result = await checkUserAuth();
                if(result?.isAuthenticated){
                    setUser(result.user);
                }
                else{
                    clearUser();
                }
            } catch (error) {
                console.error(error);
                clearUser();
            }finally{
                setIsChecking(false);
            }
        }
        varifyAuth();
    },[setUser,clearUser])

    if(isChecking){
        return <Loader/>
    }
    if(!isAuthenticated){
        return <Navigate to="/user-login" state={{from:location}} replace/>
    }

    //user is auth -render the protected route
    return <Outlet/>
}

export const PublicRoute = ()=> {
    const isAuthenticated = useUserStore(state => state.isAuthenticated);
    if(isAuthenticated){
        return <Navigate to ="/" replace />
    }
    return <Outlet/>
}